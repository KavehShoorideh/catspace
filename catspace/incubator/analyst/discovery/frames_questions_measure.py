"""PLAN 78.57 F: THE MEASURE OF THE FRAME READ BY QUESTIONS, the 78.55 G again (frames_measure.py): the same seeds, the same rows, the old numbers beside the new. A measurement in a sandbox; nothing is
wired into any player, nothing is written to rules/learned or data/derived/snapshots; no oracle, no human game, no puzzle. Game-free: no unit, place or kind of any game is named here; a kind is a word read
from the graph at run time. No worth table.

STATED BEFORE THE RUN (and in docs/ANALYST_PLAN.md, "78.57 FUSES"): the seeds are the 78.55 seeds (the top SEEDS = 12 distinct kept concepts of 78.54 by |z| on half A), the rows the quiet positions of the
test half T, the games the same 500. The fuses, named, with their reasons (costs, not tuned to a result): FUSE = 1500 questions per reading and FRAME_FUSE = 8000 questions over all the readings of one frame
(questions.rs; a line the fuse ends is UNREAD, never a verdict); ROW_FUSE = 30000 questions over all frames of one row; the depth is DEPTH = 4 plies, QUIET_PLIES = 2 quiet plies after a taking before only
takings are read on, EXT = 2 more plies of takings (questions.rs, stated there with the reason); TRAP_FUSE = 400 questions per unit; the composition test is run on every COMPOSE_EVERY-th row that has two
frames apart, with the same fuse. The frame classes are those of 78.55 (LOST, SAFE, BOTH, THREAT, CONDITIONAL, CONTESTED, UNREAD) made from the claims of the questions read (only for members that stand
attacked, or that the frame's own question names: see questions.rs claims_q).

  .venv/bin/python -m catspace.incubator.analyst.discovery.frames_questions_measure --game NAME --games-dir D --reads R [--concepts-dir C] [--games N] [--threads K] [--out-dir O]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import forcing_arrival_measure as FA
from catspace.incubator.analyst.discovery import forcing_opportunity_measure as FO
from catspace.incubator.analyst.discovery import frames_measure as OLD
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import subgoal_fit_measure as SF
from catspace.io import paths

HORIZONS = OLD.HORIZONS
SEEDS = OLD.SEEDS
MAX_SEED_SIZE = OLD.MAX_SEED_SIZE
PROFILE_GAMES = OLD.PROFILE_GAMES
FUSE = 1_500
ROW_FUSE = 30_000
COMPOSE_EVERY = 5
TRAP_FUSE = 400
CLASSES = OLD.CLASSES + ["READ_BOTH", "READ_LOST", "READ_SAFE"]       # 78.58: a claim that rests on a bound, a prune or an estimate is READ, not forced
PRIOR_DIR = ("frames_78_57", "short40b")
TRAP_WORDS = ["free", "trapped", "immobile", "not_quiet", "ended", "unread"]
OLD_DIR = ("frames_78_55", "games_s5016_n500")


def rate(v: list) -> float:
    return v[1] / v[0] if v[0] else float("nan")


def totals(obs: dict, prefix: str, h: int) -> list:
    tot = [0, 0, 0, 0]
    for k, v in obs.items():
        p = k.split("|")
        if p[0] == prefix and int(p[-1]) == h:
            tot = [x + y for x, y in zip(tot, v)]
    return tot


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", required=True)
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--concepts-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--games", type=int, default=0, help="a subset: the first N games by id")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--estimator", default=None, help="78.58: the estimator record: its kind_values are the worth of the kinds the exchange reads; without it the read is the 78.57 read")
    ap.add_argument("--prior", default=None, help="the directory of the 78.57 raw_counts.json to set beside (default frames_78_57/short40b)")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    records = sorted(FO.read_games(a.games_dir), key=lambda r: r["game"])
    variant = records[0]["variant"]
    label = Path(a.games_dir).name + (f"_first{a.games}" if a.games else "")
    out = Path(a.out_dir or paths.experiment("frames_78_57", label))
    out.mkdir(parents=True, exist_ok=True)
    concepts_dir = Path(a.concepts_dir or paths.experiment("unit_fates_78_54", Path(a.games_dir).name))
    reads_by_id = {}
    with open(a.reads) as f:
        for line in f:
            g = json.loads(line)
            reads_by_id[g["game"]] = g["plies"]
    if a.games:
        records = records[:a.games]
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS frames-q {label}: {msg}")
        print(lines[-1], flush=True)

    ids, gix, plies = [], [], []
    for g, r in enumerate(records):
        pl = reads_by_id[r["game"]]
        own, oth = FA.counts_of_game(pl)
        for t in FA.quiet_indices(pl, own, oth):
            ids.append(r["game"]); gix.append(g); plies.append(t)
    t = {"id": np.array(ids), "game": np.array(gix), "ply": np.array(plies)}
    _fit, test = SF.split_rows(t)
    say(f"games {len(records)}, quiet rows {len(ids)}, test half T (odd ids) {len(test)} rows")
    picked = OLD.pick_concepts(concepts_dir, a.seeds)
    say(f"SEEDS (the 78.55 seeds): the top {a.seeds} distinct kept concepts of 78.54 by |z| on half A; found {len(picked)}")
    model = GL.model(a.game, variant)
    kinds = [n[len("kind:"):] for ty, n in model.static_part()["nodes"] if ty == "kind"]
    wires = [(r.get("start"), list(r["moves"])) for r in records]
    model.learn_profile(wires[:PROFILE_GAMES])
    goal = G2M.goal_kind_of(variant)
    if a.estimator:
        rec = json.loads(Path(a.estimator).read_text())
        model.set_kind_values([(k, int(v)) for k, v in rec["kind_values"].items()])
        say(f"estimator record {a.estimator}: source {rec['source']}; dated {rec['date']}; the worth of the kinds the exchange reads {rec['kind_values']}")
    gi, rows = G2M.layout(t, test)
    t1 = time.perf_counter()
    res = model.questions_measure([wires[g] for g in gi], rows, [k for _d, k, _x, _z in picked], goal, list(HORIZONS), FUSE, ROW_FUSE, COMPOSE_EVERY, MAX_SEED_SIZE, True, TRAP_FUSE, a.threads)
    wall = time.perf_counter() - t1
    c, s, obs = res["count"], res["secs"], res["obs"]
    say(f"run: {wall:.1f} s wall over {a.threads} threads")
    old_dir = Path(paths.experiment(*OLD_DIR))
    old = json.loads((old_dir / "raw_counts.json").read_text()) if (old_dir / "raw_counts.json").exists() else None
    oc, os_, oobs = (old["count"], old["secs"], old["obs"]) if old else ({}, {}, {})
    pdir = Path(a.prior) if a.prior else Path(paths.experiment(*PRIOR_DIR))
    prior = json.loads((pdir / "raw_counts.json").read_text()) if (pdir / "raw_counts.json").exists() else None
    pc, ps, pobs = (prior["count"], prior["secs"], prior["obs"]) if prior else ({}, {}, {})
    pn = lambda k: pc.get(k, 0)
    n = lambda k: c.get(k, 0)
    on = lambda k: oc.get(k, 0)

    # ---- (1) cost: the number of questions per frame, beside the 78.55 positions per frame
    fr, ofr = max(1, n("frames_read")), max(1, on("frames_read"))
    say(f"1 cost: rows {n('rows')}, rows with a concept match {n('rows_with_seeds')}, matches {n('seeds')}, frames made {n('frames')} (joins refused for size and split instead {n('split_joins')}, frames dropped for a shared unit {n('dropped_shared')}), "
        f"frames read {n('frames_read')}, frames split at the root {n('frames_split_at_root')}, skipped by the row fuse {n('frames_skipped_by_row_fuse')}")
    say(f"1 cost: refused {({k.split('|', 1)[1]: v for k, v in c.items() if k.startswith('refused|')} or 0)}; members per frame read {n('members') / fr:.2f}")
    say(f"1 cost: QUESTIONS per frame {n('questions') / fr:.1f} and positions made per frame {n('positions') / fr:.1f}  |  78.55: positions read per frame {on('nodes') / ofr:.1f}, members per frame {on('members') / ofr:.2f}")
    hist = [(k.split("|", 1)[1], v) for k, v in sorted(c.items()) if k.startswith("hist|")]
    order = ["q<=1", "q<=3", "q<=10", "q<=30", "q<=100", "q<=300", "q<=1000", "q>1000"]
    say("1 cost: frames by questions asked: " + ", ".join(f"{k} {dict(hist).get(k, 0)} ({dict(hist).get(k, 0) / fr:.3f})" for k in order))
    few = lambda f, cc: sum(cc.get(f"hist|{k}", 0) for k in ("q<=1", "q<=3", "q<=10", "q<=30")) / max(1, f("frames_read"))
    pfr = max(1, pn("frames_read"))
    say(f"1 cost: share of frames at 30 questions or fewer {few(n, c):.4f} ({sum(c.get(f'hist|{k}', 0) for k in ('q<=1', 'q<=3', 'q<=10', 'q<=30'))} of {n('frames_read')})  |  78.57: {few(pn, pc):.4f}; 78.57 QUESTIONS per frame {pn('questions') / pfr:.1f}, members per frame {pn('members') / pfr:.2f}, frames read {pn('frames_read')}")
    say(f"1 cost: 78.58 exchange: takings pruned by the estimate {n('est_pruned')}, takings whose exchange was read by the estimate as one question {n('est_read')} (the line went on below, a bearing off the place changed: {n('est_cont')}), lines the game could not play {n('est_unplayed')}")
    say(f"1 cost: candidates asked {n('candidates')}, legal moves not asked {n('not_asked')}, pruned (a member lost, none gained, against one reply): the moved unit {n('pruned_moved')}, another member {n('pruned_other')}; "
        f"questions that stopped at the first move that holds {n('stops')}; lines ended by a bound {n('bounded')}; moves of a member to a place off the root frame, followed {n('follows')}")
    say(f"1 cost: frames with some line UNREAD (fuse) {n('frames_with_unread_lines')} of {n('frames_read')} ({n('frames_with_unread_lines') / fr:.4f}); lines unread {n('unread_lines')}; stuck {n('stuck')}; forced passes {n('forced_pass')}; "
        f"passes not realisable {n('pass_unrealisable')}; passes the game refused (a neutral move stood for them) {n('pass_refused')}; frames past 16 members at a node {n('too_big')}; frame gone (every seed unit taken) {n('gone')}; units read as OTHER {n('others')}")
    sec_new = s.get("seconds_seeds", 0) + s.get("seconds_frames_build", 0) + s.get("seconds_read", 0) + s.get("seconds_trap", 0)
    sec_old = os_.get("seconds_seeds", 0) + os_.get("seconds_frames_build", 0) + os_.get("seconds_read", 0)
    say(f"1 cost: TIME per frame read {1e6 * s.get('seconds_read', 0) / fr:.0f} microseconds, per position made {1e6 * s.get('seconds_read', 0) / max(1, n('positions')):.2f} us, per question {1e6 * s.get('seconds_read', 0) / max(1, n('questions')):.2f} us, "
        f"per row with a match {1e6 * (s.get('seconds_seeds', 0) + s.get('seconds_frames_build', 0) + s.get('seconds_read', 0)) / max(1, n('rows_with_seeds')):.0f} us (the trap questions apart: {s.get('seconds_trap', 0):.1f} s summed)")
    if old:
        say(f"1 cost: 78.55: per frame read {1e6 * os_.get('seconds_read', 0) / ofr:.0f} us, per position {1e6 * os_.get('seconds_read', 0) / max(1, on('nodes')):.2f} us, per row with a match {1e6 * sec_old / max(1, on('rows_with_seeds')):.0f} us; frames read {on('frames_read')}, refused for size {on('refused|too large')}")
    say(f"1 cost: arrivals tried (conditional claims) {n('arrivals_tried')}; root candidates pruned {n('root_prunes')}")

    # ---- (1b) where the questions go: by the class of the move that led to the node, by ply below the root, by frame size, small frames against big ones
    cls = ["taking", "answer (a threatened member steps away)", "adds a bearing", "new threat", "block or arrival", "root or pass"]
    gcls = cls[:5]
    for b in ("all", "small", "big", "mid"):
        fb = n(f"split|{b}|frames")
        if not fb:
            continue
        qb = n(f"split|{b}|questions")
        say(f"1b split [{b}]: frames {fb}, questions {qb} ({qb / fb:.1f} per frame), members per frame {n(f'split|{b}|members') / fb:.2f}, positions made per frame {n(f'split|{b}|positions') / fb:.1f}")
        say(f"1b split [{b}]: questions by the class of the move that led to the node: " + "; ".join(f"{cls[i]} {n(f'split|{b}|qclass|{i}')} ({n(f'split|{b}|qclass|{i}') / max(1, qb):.3f})" for i in range(6)))
        say(f"1b split [{b}]: candidates made / asked by class: " + "; ".join(f"{gcls[i]} {n(f'split|{b}|gen|{i}')} / {n(f'split|{b}|ask|{i}')}" for i in range(5)))
        say(f"1b split [{b}]: questions by ply below the root: " + ", ".join(f"{d}: {n(f'split|{b}|depth|{d}')} ({n(f'split|{b}|depth|{d}') / max(1, qb):.3f})" for d in range(16) if n(f"split|{b}|depth|{d}")))
    if prior:
        qa = pn("split|all|questions")
        say(f"1b 78.57 [all]: questions by the class of the move that led to the node: " + "; ".join(f"{cls[i]} {pn(f'split|all|qclass|{i}')} ({pn(f'split|all|qclass|{i}') / max(1, qa):.3f})" for i in range(6)))
        say(f"1b 78.57 [all]: questions by ply below the root: " + ", ".join(f"{d}: {pn(f'split|all|depth|{d}')} ({pn(f'split|all|depth|{d}') / max(1, qa):.3f})" for d in range(16) if pn(f"split|all|depth|{d}")))
    sizes = sorted({int(k.split("|")[1]) for k in c if k.startswith("size|")})
    say("1b by frame size (members: frames, questions per frame): " + "; ".join(f"{m}: {n(f'size|{m}|frames')}, {n(f'size|{m}|questions') / max(1, n(f'size|{m}|frames')):.0f}" for m in sizes))

    # ---- (2) share of member units per class, observed fates per class
    def expected(cls: str, h: int, o: dict) -> "tuple[float, float]":
        n_tot, e_taken, e_free = 0, 0.0, 0.0
        for k, v in o.items():
            p = k.split("|")
            if p[0] != cls or int(p[-1]) != h:
                continue
            b = o.get(f"ALL|{p[1]}|{p[2]}|{h}")
            if not b or not b[0]:
                continue
            n_tot += v[0]; e_taken += v[0] * b[1] / b[0]; e_free += v[0] * b[2] / b[0]
        return (e_taken / n_tot, e_free / n_tot) if n_tot else (float("nan"), float("nan"))

    table = []
    for h in HORIZONS:
        base, obase = totals(obs, "ALL", h), totals(oobs, "ALL", h)
        pbase, pmem = totals(pobs, "ALL", h), totals(pobs, "MEMBER", h)
        say(f"2 base h={h}: units {base[0]}, taken {rate(base):.4f}, taken for nothing {base[2] / max(1, base[0]):.4f}, after taking {base[3] / max(1, base[0]):.4f}  |  78.55 base: units {obase[0]}, taken {rate(obase):.4f}, for nothing {obase[2] / max(1, obase[0]):.4f}")
        mem, omem = totals(obs, "MEMBER", h), totals(oobs, "MEMBER", h)
        for cls in ["MEMBER"] + CLASSES:
            v, ov, pv = totals(obs, cls, h), totals(oobs, cls, h), totals(pobs, cls, h)
            if not v[0] and not ov[0] and not pv[0]:
                continue
            pet, pef = expected(cls, h, pobs)
            say(f"2 class {cls:<11} h={h} [78.57, 40 games]: units {pv[0]:>7} (share {pv[0] / max(1, pmem[0]):.4f}), taken {rate(pv):.4f} (base {pet:.4f}), for nothing {pv[2] / max(1, pv[0]):.4f} (base {pef:.4f})")
            et, ef = expected(cls, h, obs)
            oet, oef = expected(cls, h, oobs)
            share = v[0] / max(1, mem[0]); oshare = ov[0] / max(1, omem[0])
            say(f"2 class {cls:<11} h={h}: units {v[0]:>7} (share of member units {share:.4f}), taken {rate(v):.4f} (base for the same kinds and owners {et:.4f}), for nothing {v[2] / max(1, v[0]):.4f} (base {ef:.4f}), after taking {v[3] / max(1, v[0]):.4f}"
                f"  |  78.55: units {ov[0]:>7} (share {oshare:.4f}), taken {rate(ov):.4f} (base {oet:.4f}), for nothing {ov[2] / max(1, ov[0]):.4f} (base {oef:.4f})")
            table.append((cls, h, v, et, ef, ov, oet, oef))
    for cls in ("LOST", "THREAT"):
        m = n(f"marks|{cls}|n")
        if m:
            say(f"2 marks on {cls} claims (a mark is on a claim only if every line that supports it has it): n {m}, followed off the frame {n(f'marks|{cls}|followed') / m:.3f}, ended by a bound {n(f'marks|{cls}|bound') / m:.3f}, "
                f"a pure-loss candidate dropped on the way {n(f'marks|{cls}|prune_used') / m:.3f}")

    # ---- (3) conditional claims
    for h in HORIZONS:
        for when in ("now", "next", "no"):
            v, ov = obs.get(f"CONDCLAIM|{when}|{h}", [0, 0, 0, 0]), oobs.get(f"CONDCLAIM|{when}|{h}", [0, 0, 0, 0])
            if v[0] or ov[0]:
                say(f"3 conditional claims h={h}, the named change played {when}: claims {v[0]}, the unit taken {v[1] / max(1, v[0]):.4f}, for nothing {v[2] / max(1, v[0]):.4f}, after taking {v[3] / max(1, v[0]):.4f}  |  78.55: claims {ov[0]}")
    say(f"3 conditional claims: class CONDITIONAL units h=4: {totals(obs, 'CONDITIONAL', 4)[0]} (78.55: {totals(oobs, 'CONDITIONAL', 4)[0]}); arrivals tried {n('arrivals_tried')} (78.55: {on('arrivals_tried')})")

    # ---- (4) composition with the tempo charged
    f = {k[2:]: v for k, v in c.items() if k.startswith("F|")}
    of = {k[2:]: v for k, v in oc.items() if k.startswith("F|")}
    say(f"4 composition (tempo charged, the side to move) against the joint read: rows with two frames apart {n('rows_with_two_frames')}; " + "; ".join(f"{k} {v}" for k, v in sorted(f.items())))
    say(f"4 composition 78.55 (no tempo charge, both readings): rows with two frames apart {on('rows_with_two_frames')}; " + "; ".join(f"{k} {v}" for k, v in sorted(of.items())))

    # ---- (5) pruned moves reopened by claims of other frames
    say(f"5 pruned root moves {n('root_prunes')}; links to a claim of another frame {n('reopen|links')}: reopened {n('reopen|reopened')}, stay pruned {n('reopen|not reopened')}; "
        + "; ".join(f"{k.split('|', 2)[2]} {v}" for k, v in sorted(c.items()) if k.startswith("reopen|why|")))

    # ---- (6) the trap question
    nt = max(1, n("trap_asked"))
    say(f"6 trap: asked {n('trap_asked')} (once per member unit of every frame read); " + "; ".join(f"{w} {n('trap|' + w)} ({n('trap|' + w) / nt:.3f})" for w in TRAP_WORDS if n("trap|" + w)) +
        f"; questions per trap {n('trap_questions') / nt:.1f}, positions per trap {n('trap_positions') / nt:.1f}, {s.get('seconds_trap', 0):.1f} s summed")
    for h in HORIZONS:
        for w in TRAP_WORDS:
            v = totals(obs, f"TRAP:{w}", h)
            if v[0]:
                et = 0.0
                n_tot = 0
                for k, vv in obs.items():
                    p = k.split("|")
                    if p[0] == f"TRAP:{w}" and int(p[-1]) == h:
                        b = obs.get(f"ALL|{p[1]}|{p[2]}|{h}")
                        if b and b[0]:
                            n_tot += vv[0]; et += vv[0] * b[1] / b[0]
                say(f"6 trap {w:<10} h={h}: units {v[0]:>7}, taken {rate(v):.4f} (base for the same kinds and owners {et / max(1, n_tot):.4f}), for nothing {v[2] / max(1, v[0]):.4f}, after taking {v[3] / max(1, v[0]):.4f}")

    (out / "dump.txt").write_text("\n".join(res["dump"]) + "\n")
    say(f"dump: {len(res['dump'])} composition disagreements and reopen links kept in {out / 'dump.txt'}")
    with open(out / "class_table.tsv", "w") as fo:
        fo.write("class\thorizon\tunits\ttaken\ttaken_for_nothing\ttaken_after_taking\tbase_taken\tbase_for_nothing\told_units\told_taken\told_for_nothing\n")
        for cls, h, v, et, ef, ov, oet, oef in table:
            fo.write(f"{cls}\t{h}\t{v[0]}\t{v[1]}\t{v[2]}\t{v[3]}\t{et:.5f}\t{ef:.5f}\t{ov[0]}\t{ov[1]}\t{ov[2]}\n")
    (out / "raw_counts.json").write_text(json.dumps({"count": c, "secs": s, "obs": obs}, sort_keys=True))
    say(f"done in {time.perf_counter() - t0:.1f} s; kinds {kinds}")
    (out / "obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
