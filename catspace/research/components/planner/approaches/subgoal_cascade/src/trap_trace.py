"""catspace/planner/trap_trace.py -- the traced trap planner: the engine's first
full recognize -> verify -> commit loop, built so the per-move TRACE is the
product (Kaveh 2026-07-30: explanations may be wrong in an honest way — "I
thought this was a trap; verification says it isn't" is the loop working).

Per our-move:
  RECOGNIZE  memory: CheckpointBank.query(phi(s)) -> candidate trap structures
             (exemplar position, neighbour agreement, timing, victim side)
  VERIFY     the literal position, 1-ply v0 (stated in the trace): a candidate
             is CONFIRMED iff some legal move makes PROGRESS toward the trap
             (embedding similarity to the exemplar rises) at SOUNDNESS cost
             <= eps committor vs our best move, and the trap's victim side is
             actually the opponent. Refutation reasons are explicit; with a
             small net, "no trap here" on most moves is the EXPECTED verdict.
  COMMIT     a confirmed trap -> play its progress move; none -> committor-greedy
             fallback, and the trace says so.
"""
from __future__ import annotations

import numpy as np

EPS_SOUND = 0.05          # soundness floor: max committor concession for a trap move


class TrapTracePlanner:
    def __init__(self, bank, embed_fn, committor_fn, eps: float = EPS_SOUND):
        """embed_fn(boards)->(B,d) unnormalized; committor_fn(boards)->(B,) white-POV."""
        from catspace.research.tools.chess_specific.memo import BoundedMemo
        self.bank = bank; self.embed = embed_fn; self.committor = committor_fn
        self.eps = eps
        # cross-move memoization: re-roots and transpositions re-see positions
        self.emb_memo = BoundedMemo(100_000)
        self.c_memo = BoundedMemo(100_000)

    def _embed_memo(self, boards):
        by_fen = {b.fen(): b for b in boards}
        rows = self.emb_memo.batch_get_or(
            [b.fen() for b in boards],
            lambda miss: list(self.embed([by_fen[f] for f in miss])))
        return np.stack(rows)

    def _committor_memo(self, boards):
        by_fen = {b.fen(): b for b in boards}
        rows = self.c_memo.batch_get_or(
            [b.fen() for b in boards],
            lambda miss: list(np.asarray(self.committor([by_fen[f] for f in miss]))))
        return np.asarray(rows)

    def decide(self, board):
        import chess
        our_white = board.turn == chess.WHITE
        moves = list(board.legal_moves)
        succ = []
        for m in moves:
            b2 = board.copy(stack=False); b2.push(m); succ.append(b2)
        E = self._embed_memo([board] + succ)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        c_all = self._committor_memo([board] + succ)
        c_now = float(c_all[0]); c_succ = c_all[1:]
        c_mover = c_succ if our_white else 1 - c_succ          # our POV
        best_val = float(c_mover.max())

        cands = self.bank.query(E[0], k=64, top_traps=3, victim_white=not our_white)
        trace = dict(fen=board.fen(), committor_now=c_now, our_white=our_white,
                     verification=dict(
                         method="1-ply",
                         checks=["side: historical victims are the opponent's color",
                                 "progress: some legal move raises embedding "
                                 "similarity to the exemplar",
                                 f"soundness: that move concedes <= {self.eps:.2f} "
                                 f"committor vs our best"],
                         caveat="approachability only — does NOT prove the trap "
                                "springs; deep goal-conditioned verification is "
                                "the planned upgrade"),
                     candidates=[], decision=None)
        chosen = None
        for cand in cands:
            cand["exp_plies"] = int(round(2 * cand["med_gap"]))   # decisions -> plies
            v = dict(verdict="REFUTED", reason="", best_move=None)
            if cand["victim_white"] == our_white:
                v["reason"] = ("wrong-side trap: the victims here were "
                               f"{'White' if cand['victim_white'] else 'Black'} — us")
            else:
                import chess as _c
                e_trap = self._embed_memo([_c.Board(cand["exemplar_fen"])])[0]
                e_trap = e_trap / (np.linalg.norm(e_trap) + 1e-9)
                sim_now = float(E[0] @ e_trap)
                sims = E[1:] @ e_trap
                sound = c_mover >= best_val - self.eps
                progress = sims > sim_now
                ok = sound & progress
                v.update(sim_now=round(sim_now, 3),
                         best_sim_gain=round(float(sims.max() - sim_now), 4),
                         n_progress_moves=int(progress.sum()),
                         cheapest_approach_concession=round(
                             float(best_val - c_mover[progress].max()), 4)
                         if progress.any() else None)
                if not progress.any():
                    v["reason"] = ("no legal move approaches the structure "
                                   f"(best similarity gain {v['best_sim_gain']:+.4f})")
                elif not ok.any():
                    v["reason"] = (f"approach exists ({int(progress.sum())} moves) but "
                                   f"cheapest concedes "
                                   f"{v['cheapest_approach_concession']:.3f} committor "
                                   f"> {self.eps:.2f} soundness floor")
                else:
                    i = int(np.flatnonzero(ok)[np.argmax(sims[ok])])
                    v.update(verdict="CONFIRMED", reason="progress at sound cost",
                             best_move=moves[i].uci(),
                             concession=round(best_val - float(c_mover[i]), 4))
                    if chosen is None:
                        chosen = (i, cand)
            trace["candidates"].append({**{k: cand[k] for k in
                                           ("exemplar_fen", "support", "agreement",
                                            "med_gap", "exp_plies", "med_delta")},
                                        "verify": v})
        trace["cache"] = dict(emb_hit_rate=round(self.emb_memo.rate, 3),
                              committor_hit_rate=round(self.c_memo.rate, 3))
        if chosen is not None:
            i, cand = chosen
            trace["decision"] = dict(source="trap", move=moves[i].uci(),
                                     target=cand["exemplar_fen"],
                                     rationale=f"steering toward a trap seen "
                                               f"{cand['support']}x in similar positions, "
                                               f"expected ~{cand['exp_plies']} plies out "
                                               f"({cand['med_gap']:.0f} opponent decisions)")
            return moves[i], trace
        i = int(np.argmax(c_mover))
        trace["decision"] = dict(source="value-fallback", move=moves[i].uci(),
                                 rationale="no trap here (all candidates refuted "
                                           "or none retrieved) — playing the "
                                           "committor-best move")
        return moves[i], trace
