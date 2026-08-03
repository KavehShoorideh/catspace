#!/usr/bin/env python
"""bootstrap_mate experiment driver -- at what search budget does KRRvK-central hit 100%?

The engine glue itself is shipping code in ../src/engine.py; this file is the sweep that
answers the budget question and is NOT packaged.

Fast path: priors cached by position (stable net), F-embeddings cached by position (stable
towers), only the bank-min-distance recomputed as the bank grows (cheap). Field net on MPS.
Games parallelized across workers sharing discoveries via an append-only FEN file.

Run (launcher spawns workers):   --nodes 5000 --n 48 --j 4
Single worker (internal):        --nodes 5000 --n 48 --j 4 --worker 0
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

from catspace.approaches.bootstrap_mate.src.engine import (MilestoneCache, OnlineMateBank,
                                                           harvest, make_batched_energy_prior,
                                                           make_boot_value, make_planner,
                                                           tb_white_move)
from catspace.fields import FieldModel
from catspace.research.components.memory.approaches.experience_store.src.experience import ExperienceStore
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, tb_best_move
from catspace.research.components.search.approaches.puct_mcts.src.mcts import MCTS
from experiments.mate_ladder_eval import sample_scenarios


def gen_7p_starts(rng, n, sf, min_cp=600):
    """KRRvKBNP (7 pieces, BEYOND the tablebase): random legal White-to-move positions
    certified decisively winning by deep SF eval (referee role, offline -- like tb
    certification, never consulted at play). The exam: navigate 7 -> trade-down -> <=6
    (tb defense resumes) -> nucleus -> mate."""
    out = []
    while len(out) < n:
        sqs = rng.choice(64, size=7, replace=False)
        b = chess.Board(None)
        for sq, (pt, col) in zip(sqs, [(chess.KING, True), (chess.ROOK, True), (chess.ROOK, True),
                                       (chess.KING, False), (chess.BISHOP, False),
                                       (chess.KNIGHT, False), (chess.PAWN, False)]):
            if pt == chess.PAWN and chess.square_rank(int(sq)) in (0, 7):
                break
            b.set_piece_at(int(sq), chess.Piece(pt, col))
        else:
            b.turn = chess.WHITE
            if b.is_valid() and not b.is_game_over():
                info = sf.analyse(b, chess.engine.Limit(depth=12))
                sc = info["score"].white().score(mate_score=10000)
                if sc is not None and sc >= min_cp:
                    out.append(b)
    return out


def worker(args):
    t0 = time.time(); tb = TB()
    sf_def = None
    if args.fen_file:
        starts = [chess.Board(f) for f in Path(args.fen_file).read_text().splitlines()
                  if f.strip()][:args.n]
    elif args.scenario == "fullgame":
        # THE RESEARCH FRONTIER (Kaveh 2026-07-25: 'end-to-end full game'): standard
        # starts, human-proxy opponent (maia, sampled), planner + tactics + probes live,
        # tb only as the <=6p logged fallback. Toy scenarios are integration tests now.
        starts = [chess.Board() for _ in range(args.n)]
        sf_def = chess.engine.SimpleEngine.popen_uci(
            ["lc0", f"--weights={args.opponent_weights}"])
    elif args.scenario == "KRRvKBNP-7p":
        sf_def = chess.engine.SimpleEngine.popen_uci(["stockfish"])
        starts = gen_7p_starts(np.random.default_rng(args.seed), args.n, sf_def)
    else:
        starts = dict(sample_scenarios(np.random.default_rng(args.seed), args.n))[args.scenario]
    fm = FieldModel(args.field, device=args.device)
    bank = OnlineMateBank(fm, Path(args.bank_file))
    loss_bank = OnlineMateBank(fm, Path(args.loss_bank_file))
    draw_bank = OnlineMateBank(fm, Path(args.draw_bank_file))
    game_ctx: dict = {}
    times: dict = {}
    vfn = make_boot_value(fm, bank, times, loss_bank,
                          dtm_ckpt=args.last_mile_dtm or None,
                          nucleus_max_pieces=args.nucleus_max_pieces,
                          draw_bank=draw_bank, game_ctx=game_ctx)
    pfn, pfnb = make_batched_energy_prior(args.energy_ckpt, device="cpu", times=times,
                                          game_ctx=game_ctx, plan_alpha=args.plan_alpha)
    planner = make_planner(fm, bank, loss_bank=loss_bank, draw_bank=draw_bank,
                           game_ctx=game_ctx, prior_fn=pfn)
    ms = MilestoneCache(fm, Path(args.milestone_file))
    exp = ExperienceStore(args.experience_db) if args.experience_db else None
    try:
        eng_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True).stdout.strip()
    except Exception:
        eng_commit = ""
    from catspace.introspection import ProbeKit
    probes = ProbeKit(fm, bank, loss_bank, draw_bank,
                      exp_db=(exp.db if exp is not None else None),
                      game_ctx=game_ctx, prior_fn=pfn)

    res_path = Path(args.results_file)
    done = set()
    if res_path.exists():
        import json
        done = {json.loads(ln)["g"] for ln in res_path.read_text().splitlines() if ln.strip()}
    base = ([int(x) for x in args.games.split(",")] if args.games
            else list(range(len(starts))))
    my_games = [g for g in base[args.worker::args.j] if g not in done]
    if done:
        print(f"[worker {args.worker}] resume: skipping {len(done)} recorded games", flush=True)

    def _stale() -> bool:
        """ALWAYS-RUN-LATEST enforcement (Kaveh: 'stale tests shouldn't run'; 2026-07-23:
        'don't kill the workers on a commit -- have them reload the newest and work on
        that'): if HEAD moved since this process launched, the worker re-execs ITSELF
        (same argv, new code image) at the next game boundary -- resume skips recorded
        games, the WIP checkpoint carries any in-flight game, no launcher needed."""
        try:
            now = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
            return bool(eng_commit) and bool(now) and now != eng_commit
        except Exception:
            return False

    # WORKER CHECKPOINTS (Kaveh: 'enforcement absolutely must come with worker checkpoints
    # so we can restart mistaken stops'): a WIP file per worker, written EVERY ply; a
    # relaunched worker resumes its in-flight game mid-play (board rebuilt from ucis,
    # counters restored). Deleted on game completion. Covers stale-exits, crashes, kills.
    import json as _json
    wip_path = Path(f"{args.results_file}.wip.w{args.worker}.json")
    resume_state = None
    if wip_path.exists():
        try:
            cand_state = _json.loads(wip_path.read_text())
            if cand_state.get("g") not in done and cand_state.get("g") in base:
                resume_state = cand_state
                my_games = [resume_state["g"]] + [g for g in my_games if g != resume_state["g"]]
                print(f"[worker {args.worker}] WIP checkpoint: resuming g{resume_state['g']} "
                      f"at ply {len(resume_state['ucis'])}", flush=True)
        except Exception:
            pass

    results = []
    for gi in my_games:
        if _stale():
            print(f"[worker {args.worker}] code moved past {eng_commit}; "
                  f"re-exec onto newest", flush=True)
            tb.close()
            if sf_def is not None:
                sf_def.quit()
            os.execv(sys.executable, [sys.executable] + sys.argv)
        bank.sync(); loss_bank.sync(); draw_bank.sync(); ms.sync()
        b = starts[gi].copy(stack=False)
        start_epd = b.epd()
        plies = 0; nodes_spent = 0; tmoves = []; found_this_game = 0
        roots: list[str] = []; mseen: list[bool] = []; nmoves: list[int] = []
        ucis: list[str] = []    # full trajectory (start_epd + ucis reproduces the game)
        tb_consults: list[int] = []   # plies where the tb fallback fired (attribution log)
        _rs = None
        if resume_state is not None and resume_state.get("g") == gi:
            _rs, resume_state = resume_state, None
        from collections import Counter as _Counter
        hist = _Counter({b.epd(): 1})   # ALL position visits, both colors (stuckness +
                                        # repetition-creation triggers); counted at arrival
        tb_mode = False   # STICKY handover (g036: alternating field/tb control oscillates
                          # through the consulted position into threefold; once the field
                          # proves gradient-less in a game, tb converts the rest, all logged)
        game_ctx["hist"] = hist   # live repetition counts -> kappa's moving-surface term
        game_ctx["plan"] = "direct"
        plan_counts = _Counter()
        noharvest = 0     # WITHIN-GAME progress gating (Kaveh: no cross-game self-stats):
                          # consecutive searches touching zero new mates = value failing NOW
        prev_v = None; tactic_events: list = []; probe_snaps: list = []
        # PER-RECORD PROVENANCE (Kaveh: 'data generated needs to store the commit that
        # generated it'): with in-place re-exec a game can span commits; the record
        # carries every commit that produced plies, not just the finisher's.
        game_commits = {eng_commit} if eng_commit else set()
        if _rs is not None:                     # WIP restore: replay trajectory, rebuild
            for u in _rs["ucis"]:               # hist at each arrival, restore counters
                b.push(chess.Move.from_uci(u))
                hist[b.epd()] += 1
            ucis = list(_rs["ucis"]); plies = len(ucis)
            nodes_spent = _rs.get("nodes_spent", 0); tmoves = list(_rs.get("tmoves", []))
            roots = list(_rs.get("roots", [])); mseen = list(_rs.get("mseen", []))
            nmoves = list(_rs.get("nmoves", [])); tb_consults = list(_rs.get("tb_consults", []))
            found_this_game = _rs.get("found", 0); tb_mode = _rs.get("tb_mode", False)
            noharvest = _rs.get("noharvest", 0); prev_v = _rs.get("prev_v")
            tactic_events = [tuple(t) for t in _rs.get("tactics", [])]
            probe_snaps = list(_rs.get("probes", []))
            plan_counts.update(_rs.get("plans", {}))
            game_commits.update(_rs.get("commits", []))

        def _save_wip():
            try:
                wip_path.write_text(_json.dumps(dict(
                    g=gi, start_epd=start_epd, ucis=ucis, nodes_spent=nodes_spent,
                    tmoves=tmoves, roots=roots, mseen=mseen, nmoves=nmoves,
                    tb_consults=tb_consults, found=found_this_game, tb_mode=tb_mode,
                    noharvest=noharvest, prev_v=prev_v, tactics=tactic_events,
                    probes=probe_snaps, plans=dict(plan_counts),
                    commits=sorted(game_commits)),
                    default=float))          # np scalars (prev_v!) must not kill WIP
            except Exception as e:           # NEVER silent (2026-07-24: prev_v np.float32
                if not getattr(_save_wip, "warned", False):   # broke WIP for hours --
                    _save_wip.warned = True                   # re-execs lost whole games)
                    print(f"[worker {args.worker}] WIP SAVE FAILING: {e}", flush=True)
        # TACTICS TRACKER (Kaveh; INQUIRY_TACTICS: 'a tactic is an opportunity outside our
        # plan afforded by a mistake by our opponent'): an upward DISCONTINUITY in our own
        # root value across the opponent's reply = a detected opportunity-from-mistake.
        # Within-game, own-values-only, no concepts; logged for the pounce mechanism later.
        reuse = None            # subtree carried across moves (tree reuse; general lever)
        # WARM-UP (Kaveh 2026-07-25 'run until bank is full on first move'): before the
        # first timed move, keep searching+harvesting until the bank reaches the target
        # (cap: 20x one move's budget). Warm trees feed the real move via reuse.
        if args.warm_bank > 0 and len(bank) < args.warm_bank and b.turn == chess.WHITE:
            tw = time.time(); warm_nodes = 0; warm0 = len(bank)
            while len(bank) < args.warm_bank and warm_nodes < 20 * args.nodes:
                if hasattr(vfn, "set_anchor"):
                    vfn.set_anchor(b)
                mw = MCTS(lambda bs: np.zeros(len(bs)), max_nodes=args.nodes, mate_stop=True,
                          pw_c=1.5, root_min_visits=10, value_fn=vfn, policy_fn=pfn,
                          policy_batch_fn=pfnb, batch_leaves=32)
                wroot = mw.run(b, reuse_root=reuse)
                w_w, w_l, w_s = harvest(wroot)
                bank.add(w_w)
                if w_l:
                    loss_bank.add(w_l)
                if w_s:
                    draw_bank.add(w_s)
                warm_nodes += mw.evals_used
                reuse = wroot
                if mw.evals_used <= 1:          # mate at root: no more warming needed
                    break
            print(f"    [warm-up] bank {warm0}->{len(bank)} nodes={warm_nodes} "
                  f"[{time.time()-tw:.0f}s]", flush=True)
        # asymmetric claiming (g041 harness artifact: outcome(claim_draw=True) auto-claims
        # for EITHER side -- White would never claim a draw in a won position; only the
        # DEFENDER claims). Automatic draws (75-move/fivefold/stalemate/material) still end.
        def _game_over():
            if b.is_game_over():
                return True
            return b.turn == chess.BLACK and (b.can_claim_threefold_repetition()
                                              or b.can_claim_fifty_moves())
        while plies < args.max_plies and not _game_over():
            _save_wip()
            # PLY WATCHDOG (2026-07-23 spin-bug: a technique worker spun 100% CPU for 2h
            # inside ONE ply, silently; repro with reconstructed state came back clean,
            # so the trigger needs live state we don't capture). If any single ply takes
            # 30 min, dump the ACTUAL spinning stack to the log and exit -- the chain
            # relaunches and WIP resumes; next occurrence self-diagnoses.
            import faulthandler
            faulthandler.dump_traceback_later(1800, exit=True)
            if b.turn == chess.WHITE:
                # STUCKNESS trigger (Kaveh 'do the fix'): second visit to a position =
                # the field has no EFFECTIVE gradient in play (confidently-wrong loops
                # never trip the flatness trigger) -> consult tb directly, LOGGED.
                if (args.tb_fallback_eps > 0
                        and (tb_mode or hist[b.epd()] >= 2 or b.halfmove_clock >= 60
                             or plies >= 60 or noharvest >= 6)
                        and len(b.piece_map()) <= 6):
                    mv_tb = tb_white_move(b, tb)
                    if mv_tb is not None:
                        tb_mode = True
                        tb_consults.append(plies)
                        roots.append(b.epd()); mseen.append(False); nmoves.append(0)
                        tmoves.append(0.0); ucis.append(mv_tb.uci())
                        if (b.is_capture(mv_tb) or b.piece_type_at(mv_tb.from_square) == chess.PAWN) \
                                and hasattr(vfn, "invalidate_anchor"):
                            vfn.invalidate_anchor()
                        b.push(mv_tb)
                        hist[b.epd()] += 1
                        reuse = None
                        plies += 1
                        continue
                tm = time.time(); snap = dict(times)
                m = MCTS(lambda bs: np.zeros(len(bs)), max_nodes=args.nodes, mate_stop=True,
                         pw_c=1.5, root_min_visits=10, value_fn=vfn, policy_fn=pfn,
                         policy_batch_fn=pfnb, batch_leaves=32)
                roots.append(b.epd())
                ps = planner(b, plies)
                if ps["plan"] != game_ctx.get("plan") or plies == 0:   # switch OR game open:
                    snap = probes.summary(b)               # log the planner's observation
                    probe_snaps.append(dict(ply=plies, plan=ps["plan"], **snap))
                game_ctx["plan"] = ps["plan"]; game_ctx["target_pt"] = ps.get("target_pt")
                plan_counts[ps["plan"] + (f"->{ps['goal']}" if ps.get("goal") else "")] += 1
                if hasattr(vfn, "set_anchor"):
                    vfn.set_anchor(b)
                if reuse is not None:
                    reuse.parent = None     # detach: stale ancestors skew the mate-depth
                                            # discount and double-count _threefold's walk
                root = m.run(b, reuse_root=reuse)
                t_search = time.time() - tm
                v_root = root.W / max(root.N, 1)
                if prev_v is not None and (v_root - prev_v) > 0.15:
                    tactic_events.append((plies, round(float(v_root - prev_v), 3)))
                prev_v = v_root
                th = time.perf_counter()
                win_mates, loss_mates, stales = harvest(root)
                mseen.append(len(win_mates) > 0); nmoves.append(m.evals_used)
                added = bank.add(win_mates)
                found_this_game += added
                noharvest = 0 if added > 0 else noharvest + 1
                if loss_mates:
                    loss_bank.add(loss_mates)
                if stales:
                    draw_bank.add(stales)
                t_harv = time.perf_counter() - th
                best = max(root.children,
                           key=lambda c: (c.N, (c.terminal_v if c.terminal_v is not None else c.Q)))
                # OPENING TEMPERATURE (2026-07-24: improvement-loop round 0 produced
                # IDENTICAL games per worker -- engine argmax + maia nodes=1 are both
                # deterministic, so a 10-game batch held 2 unique games). AZ convention:
                # sample the first plies from the root visit distribution, seeded per
                # (game, worker); certified mates are never sampled away.
                if (args.scenario == "fullgame" and plies < 8
                        and best.terminal_v is None and len(root.children) > 1):
                    _rng = np.random.default_rng((gi * 1009 + args.worker * 9973
                                                  + plies * 31) % 2**32)
                    ns = np.array([float(c.N) for c in root.children])
                    if ns.sum() > 0:
                        # tau=2 flatten + uniform floor: raw visit dists are peaked
                        # enough that tau=1 sampling reproduced argmax (two identical
                        # 32-ply games post-fix); this actually diversifies
                        p = np.sqrt(ns / ns.sum())
                        p = 0.8 * p / p.sum() + 0.2 / len(ns)
                        best = list(root.children)[int(_rng.choice(len(ns), p=p / p.sum()))]
                nodes_spent += m.evals_used; tmoves.append(time.time() - tm)
                # TB FALLBACK (Kaveh): if the searched root shows NO gradient (children
                # value spread < eps) and the position is tb-probeable, consult tb and LOG.
                if args.tb_fallback_eps > 0 and len(b.piece_map()) <= 6 and root.children:
                    qs = [c.terminal_v if c.terminal_v is not None
                          else (c.W / c.N if c.N > 0 else None) for c in root.children]
                    qs = [q for q in qs if q is not None]
                    if len(qs) >= 2 and (max(qs) - min(qs)) < args.tb_fallback_eps:
                        mv_tb = tb_white_move(b, tb)
                        if mv_tb is not None:
                            best = next((c for c in root.children if c.move == mv_tb), best)
                            tb_consults.append(plies)
                # repetition-CREATION veto: if the chosen move pushes into a position
                # already seen twice (threefold = instant draw claim), consult tb instead
                if args.tb_fallback_eps > 0 and len(b.piece_map()) <= 6:
                    nb2 = b.copy(stack=False); nb2.push(best.move)
                    if hist[nb2.epd()] >= 2:
                        mv_tb = tb_white_move(b, tb)
                        if mv_tb is not None and mv_tb != best.move:
                            best = next((c for c in root.children if c.move == mv_tb), best)
                            tb_mode = True
                            tb_consults.append(plies)
                d = {k: times.get(k, 0) - snap.get(k, 0) for k in
                     ("prior_s", "prior_n", "embedF_s", "embedF_n", "dbank_s", "dbank_n",
                      "dtm_s", "dtm_n")}
                tree = t_search - d["prior_s"] - d["embedF_s"] - d["dbank_s"] - d["dtm_s"]
                try:                                    # observability (Prometheus)
                    from catspace.research.tools.stats_eval.metrics import observe
                    for st, key in (("prior", "prior_s"), ("embF", "embedF_s"),
                                    ("dbank", "dbank_s"), ("dtm", "dtm_s")):
                        observe(st, d[key])
                    observe("tree", max(tree, 0)); observe("harvest", t_harv)
                    observe("move_total", t_search)
                except Exception:
                    pass
                print(f"    mv{len(tmoves):02d} {tmoves[-1]:6.1f}s = prior {d['prior_s']:5.1f} "
                      f"({d['prior_n']:4d}) + embF {d['embedF_s']:5.1f} ({d['embedF_n']:4d}) "
                      f"+ dbank {d['dbank_s']:5.1f} ({d['dbank_n']:5d}) + tree {tree:5.1f} "
                      f"+ harvest {t_harv:4.1f}  nodes={m.evals_used}", flush=True)
                # ANNOUNCE-rule guard (g017: python-chess honors FIDE's claim-by-announcing
                # -- Black can claim a threefold it never plays out; arrival counters are
                # blind to it). If the chosen move leaves Black an immediate claim, walk
                # the tb-winning moves by dtz until one is claim-safe.
                mv_final = best.move
                if args.tb_fallback_eps > 0 and len(b.piece_map()) <= 6:
                    b.push(mv_final)
                    unsafe = b.can_claim_threefold_repetition()
                    b.pop()
                    if unsafe:
                        alts = []
                        for m2 in b.legal_moves:
                            c2 = b.copy(stack=False); c2.push(m2)
                            w2, dz2 = tb.wdl_dtz(c2)
                            if w2 is not None and -w2 == 2:
                                alts.append((abs(dz2) if dz2 is not None else 999, m2))
                        for _, m2 in sorted(alts, key=lambda x: x[0]):
                            b.push(m2)
                            ok = not b.can_claim_threefold_repetition()
                            b.pop()
                            if ok:
                                mv_final = m2
                                tb_mode = True
                                tb_consults.append(plies)
                                best = next((c for c in root.children if c.move == m2), best)
                                break
                ucis.append(mv_final.uci())
                if (b.is_capture(mv_final) or b.piece_type_at(mv_final.from_square) == chess.PAWN) \
                        and hasattr(vfn, "invalidate_anchor"):
                    vfn.invalidate_anchor()      # irreversible: candidate set is void
                b.push(mv_final)
                hist[b.epd()] += 1               # count BOTH colors (route-independent
                if hist[b.epd()] >= 2:           # repetitions live on Black-side keys too)
                    tb_mode = True               # g043: BLACK completes threefolds -- any
                reuse = best if best.move == mv_final else None   # 2nd occurrence => tb
            else:
                if args.scenario == "fullgame" and sf_def is not None:
                    # maia opponent for the WHOLE game (nodes=1 = the human-move protocol)
                    mvb = sf_def.play(b, chess.engine.Limit(nodes=1)).move
                elif len(b.piece_map()) > 6 and sf_def is not None:
                    # beyond tb: STOCKFISH defends until the trade-down re-enters tb range
                    mvb = sf_def.play(b, chess.engine.Limit(nodes=20000)).move
                else:
                    mvb = tb_best_move(b, tb)
                if reuse is not None:
                    reuse = next((c for c in reuse.children if c.move == mvb), None)
                if (b.is_capture(mvb) or b.piece_type_at(mvb.from_square) == chess.PAWN) \
                        and hasattr(vfn, "invalidate_anchor"):
                    vfn.invalidate_anchor()
                ucis.append(mvb.uci())
                b.push(mvb)
                hist[b.epd()] += 1
                if hist[b.epd()] >= 2:
                    tb_mode = True
            plies += 1
        out = b.outcome(claim_draw=True)
        mated = bool(out and out.winner == chess.WHITE)
        term = out.termination.name.lower() if out else "timeout"
        wip_path.unlink(missing_ok=True)
        ms.record_game(roots, mated, mseen, nmoves)
        if exp is not None:
            exp.record_game(args.scenario, start_epd, "mate" if mated else "fail", term,
                            ucis, roots, engine_commit=",".join(sorted(game_commits)),
                            field_ckpt=args.field,
                            opponent=Path(args.opponent_weights).stem
                            if args.scenario == "fullgame" else "")
        import json
        rec = dict(g=gi, mate=mated, term=term, plies=plies, nodes=nodes_spent,
                   t=round(sum(tmoves), 1), moves=len(tmoves), bank=len(bank),
                   tb_consults=tb_consults, plans=dict(plan_counts), tactics=tactic_events,
                   probes=probe_snaps, commits=sorted(game_commits))
        if not mated:           # FAILs carry the full trajectory for field diagnostics
            rec["start_epd"] = start_epd; rec["ucis"] = ucis
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        results.append((gi, mated, plies, nodes_spent, sum(tmoves), len(tmoves)))
        print(f"  g{gi:03d}[w{args.worker}] {'mate' if mated else 'FAIL:' + term} plies={plies} "
              f"tb={len(tb_consults)} tac={len(tactic_events)} "
              f"plan={','.join(f'{k}:{v}' for k, v in plan_counts.items())} "
              f"bank={len(bank)}(+{found_this_game}) "
              f"loss={len(loss_bank)} draws={len(draw_bank)} ms={len(ms.stats)} "
              f"t/move={np.median(tmoves):.1f}s t/game={sum(tmoves):.0f}s "
              f"nodes/s={nodes_spent/max(sum(tmoves),1e-9):.0f} [{time.time()-t0:.0f}s]", flush=True)
    tb.close()
    m_ = [r for r in results if r[1]]
    print(f"[worker {args.worker}] {len(m_)}/{len(results)} mate  "
          f"med t/move={np.median([t/max(k,1) for _, _, _, _, t, k in results]):.1f}s", flush=True)
    if sf_def is not None:
        sf_def.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--j", type=int, default=4)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--scenario", default="KRRvK-central")
    ap.add_argument("--field", default="data/derived/sep/lichess_mc2.pt")
    ap.add_argument("--energy-ckpt", default="data/derived/sep/opponent_energy_v1.pt")
    ap.add_argument("--bank-file", default=None)
    ap.add_argument("--loss-bank-file", default=None)
    ap.add_argument("--draw-bank-file", default=None)
    ap.add_argument("--milestone-file", default=None)
    ap.add_argument("--results-file", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="wipe bank/milestones/results; DEFAULT resumes (checkpointed runs)")
    ap.add_argument("--games", default=None,
                    help="comma list of game indices to (re)play (default: all 0..n-1)")
    ap.add_argument("--fen-file", default=None,
                    help="explicit start positions, one FEN per line (integration tests)")
    ap.add_argument("--opponent-weights", default="data/engines/maia/maia-1500.pb.gz",
                    help="fullgame opponent net (maia elo ladder / real lc0)")
    ap.add_argument("--warm-bank", type=int, default=1000,
                    help="first-move warm-up: search+harvest until the bank has this many "
                         "mates (cap 20x --nodes); 0 = off")
    ap.add_argument("--last-mile-dtm", default="data/derived/sep/dtm_tok_r3.pt",
                    help="tb-trained DTM regression as the value's distance source INSIDE "
                         "the nucleus (resignation gap: the field has no trajectory support "
                         "there); '' = off")
    ap.add_argument("--nucleus-max-pieces", type=int, default=6,
                    help="DTM-net value inside <=N pieces. Boundary follows the net's "
                         "training support: was 5 (CNN era, 5p-only data); dtm_tok trains "
                         "on ALL tb classes incl 6-man, so 6 (KRRvKBP autopsy: all FAILs "
                         "threw the tb win in the first 4 moves -- 6p starts sat OUTSIDE "
                         "the nucleus on an empty bank in resignation-gap field regions)")
    ap.add_argument("--plan-alpha", type=float, default=1.0,
                    help="planner's prior-bias strength (alpha-dial; 0 = planner off)")
    ap.add_argument("--experience-db", default="data/derived/experience.sqlite",
                    help="persistence layer: every game + searched roots + provenance; "
                         "'' disables")
    ap.add_argument("--import-banks", default=None,
                    help="path prefix of another run's banks (<prefix>_bank.fens / "
                         "_lossbank / _drawbank) to seed this run's banks (merged, "
                         "deduped). Banks are FACTS: they survive engine and field "
                         "changes and re-embed at load (Kaveh: build one bank, reuse)")
    ap.add_argument("--tb-fallback-eps", type=float, default=0.02,
                    help="if the searched root's child-value spread < eps (no field "
                         "gradient) and <=6 pieces: consult tb, LOG the consult (win "
                         "attribution); 0 = never consult")
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _eptr = Path("data/derived/sep/opponent_energy_current.txt")   # improvement-loop swaps
    if args.energy_ckpt == ap.get_default("energy_ckpt") and _eptr.exists():
        args.energy_ckpt = _eptr.read_text().strip()
    tag = f"n{args.nodes}_{args.scenario}"
    if args.bank_file is None:
        args.bank_file = f"artifacts/experiments/boot_bank_{tag}.fens"
    if args.loss_bank_file is None:
        args.loss_bank_file = f"artifacts/experiments/boot_lossbank_{tag}.fens"
    if args.draw_bank_file is None:
        args.draw_bank_file = f"artifacts/experiments/boot_drawbank_{tag}.fens"
    if args.milestone_file is None:
        args.milestone_file = f"artifacts/experiments/boot_milestones_{tag}.jsonl"
    if args.results_file is None:
        args.results_file = f"artifacts/experiments/boot_results_{tag}.jsonl"

    if args.worker is not None:
        worker(args); return

    if args.fresh:
        for p in (args.bank_file, args.loss_bank_file, args.draw_bank_file,
                  args.milestone_file, args.results_file):
            Path(p).unlink(missing_ok=True)
    if args.import_banks:
        for src_sfx, dst in (("_bank.fens", args.bank_file),
                             ("_lossbank.fens", args.loss_bank_file),
                             ("_drawbank.fens", args.draw_bank_file)):
            src = Path(args.import_banks + src_sfx)
            if src.exists():
                have = set(Path(dst).read_text().splitlines()) if Path(dst).exists() else set()
                new = [l for l in src.read_text().splitlines() if l.strip() and l not in have]
                with open(dst, "a") as f:
                    f.writelines(l + "\n" for l in new)
                print(f"[import] {src} -> {dst}: +{len(new)}", flush=True)
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--worker", str(w)])
             for w in range(args.j)]
    rcs = [p.wait() for p in procs]
    if any(rc == 75 for rc in rcs):
        print("[launcher] workers went STALE (code updated); exiting 75 for relaunch", flush=True)
        sys.exit(75)
    import json
    raw = [json.loads(ln) for ln in Path(args.results_file).read_text().splitlines() if ln.strip()] \
        if Path(args.results_file).exists() else []
    first = {}
    for r in raw:                       # dedup by game id, first occurrence wins (crash-replay safety)
        first.setdefault(r["g"], r)
    rows = list(first.values())
    n_bank = len(set(Path(args.bank_file).read_text().splitlines())) if Path(args.bank_file).exists() else 0
    m = [r for r in rows if r["mate"]]
    tba = [r for r in m if r.get("tb_consults")]
    tpm = [r["t"] / max(r["moves"], 1) for r in rows]
    print(f"VERDICT BOOTSTRAP_MATE scenario={args.scenario} nodes={args.nodes} "
          f"mate={len(m)}/{len(rows)} ({len(m)/max(len(rows),1):.2f}) "
          f"[clean={len(m)-len(tba)} tb-assisted={len(tba)}]  "
          f"med_plies={np.median([r['plies'] for r in m]) if m else float('nan'):.0f}  "
          f"bank_final={n_bank}  med_t/move={np.median(tpm) if tpm else float('nan'):.1f}s  "
          f"med_t/solve={np.median([r['t'] for r in m]) if m else float('nan'):.0f}s  "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
